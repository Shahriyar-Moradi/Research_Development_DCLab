"""Package 34.13 through the studio: would more rows help this project's model? A separate analysis on the training folds
of a project whose models stage ran; the held-back cases are never read, no stage record changes, and the answer is
captured as the ledger point ``data.more_rows`` and a ``decision.proposed`` event."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd

    from dclab_rnd.kinds import acceptance
except ImportError as error:  # pydantic is not part of requirements/ci.txt
    MISSING = str(error)
else:
    MISSING = ""

STAGES = ("data", "leakage", "features", "models")


def project_of(shape, seed=0):
    from dclab_core.replicas import learning_curves
    from dclab_rnd.expansion import runner
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.studio import engine

    replica = learning_curves.generate(shape, seed)
    case = acceptance.Case(replica.rows, "outcome", "binary", "When a row arrives, before its outcome is known.",
                           {"positive_label": "yes", "identifiers": ["row_id"]}, facts=replica.facts)
    store, pid = acceptance.build(case, Path(tempfile.mkdtemp()))
    with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
        for stage in STAGES:
            engine.execute(store, pid, stage)
    return store, pid


@unittest.skipIf(MISSING, MISSING)
class DataValueStudioTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd import experience, ledger

        home = Path(tempfile.mkdtemp())
        self.log, self.book = experience.open_log(home, "wtest"), ledger.FileLedger(home / "ledger", "wtest")
        experience.install(self.log)
        ledger.install(self.book)
        self.addCleanup(experience.install, None)
        self.addCleanup(ledger.install, None)

    def measure(self, store, pid):
        from dclab_rnd.models.gateway import private_scope
        from dclab_rnd.studio import data_value

        with private_scope(True):
            return data_value.measure(store, pid, bootstrap=100)

    def test_a_rising_project_hears_more_rows_likely_help_and_the_answer_is_captured(self):
        from dclab_core.ledger import check_state

        store, pid = project_of("rising")
        before = {stage: copy.deepcopy(store.read_stage(pid, stage)) for stage in STAGES}
        record = self.measure(store, pid)
        doc = record["document"]
        self.assertEqual(doc["answer"], "likely_help", doc["plain"])
        self.assertFalse(doc["holdout_read"])
        self.assertEqual(store.get(pid)["data_value"]["document"]["answer"], "likely_help")
        self.assertEqual({stage: store.read_stage(pid, stage) for stage in STAGES}, before)  # no stage record changes
        self.assertEqual(store.get(pid).get("holdout_uses", 0), 0)
        decided = [r for r in self.book.records() if r.point == "data.more_rows"]
        self.assertEqual(len(decided), 1)
        self.assertEqual((decided[0].used, decided[0].value_used), ("rule", "likely_help"))
        self.assertEqual(check_state(dict(decided[0].state)), [])
        self.assertNotIn("east", str(decided[0].state))  # a segment's level never enters a state
        events = [e for e in self.log.read(project=pid, kinds=["decision.proposed"]) if e.payload.get("point") == "data.more_rows"]
        self.assertEqual([e.payload["choice"] for e in events], ["likely_help"])

    def test_a_flat_project_hears_more_rows_are_unlikely_to_help(self):
        store, pid = project_of("flat")
        doc = self.measure(store, pid)["document"]
        self.assertEqual(doc["answer"], "unlikely_help", doc["plain"])

    def test_no_held_back_row_is_ever_read(self):
        from dclab_rnd.expansion import datasets as ds
        from dclab_rnd.expansion import runner
        from dclab_rnd.studio import engine

        store, pid = project_of("rising")
        bundle = engine.prepare(store, store.get(pid)).bundle
        held = set(bundle.X.iloc[bundle.test_idx]["row_id"])
        training = set(bundle.X.iloc[bundle.train_idx]["row_id"])
        seen = []
        real = runner.fit_and_predict

        def spy(bundle, recipe, family, params, X_fit, y_fit, X_eval, **kw):
            seen.append((set(X_fit["row_id"]), set(X_eval["row_id"])))
            return real(bundle, recipe, family, params, X_fit, y_fit, X_eval, **kw)

        def refused(*_):
            raise AssertionError("the held-back rows were read")

        with mock.patch.object(runner, "fit_and_predict", spy), \
                mock.patch.object(ds.TaskBundle, "X_test", property(refused)), mock.patch.object(ds.TaskBundle, "y_test", property(refused)):
            self.measure(store, pid)
        self.assertTrue(seen)
        for fit, evaluated in seen:
            self.assertFalse((fit | evaluated) & held)
            self.assertTrue(fit <= training and evaluated <= training)
            self.assertFalse(fit & evaluated)

    def test_a_project_without_its_models_stage_is_refused(self):
        from dclab_rnd.studio import data_value, engine
        from dclab_rnd.expansion import runner
        from dclab_rnd.models.gateway import private_scope
        from dclab_core.replicas import learning_curves

        replica = learning_curves.generate("flat", 0, n=600)
        case = acceptance.Case(replica.rows, "outcome", "binary", "When a row arrives.", {"positive_label": "yes", "identifiers": ["row_id"]})
        store, pid = acceptance.build(case, Path(tempfile.mkdtemp()))
        with runner.fast_profile(), private_scope(True):
            engine.execute(store, pid, "data")
        with self.assertRaises(data_value.DataValueError):
            data_value.measure(store, pid)



@unittest.skipIf(MISSING, MISSING)
class DataValueRouteTest(unittest.TestCase):
    def test_the_routes_refuse_before_the_models_stage_and_read_nothing_yet(self):
        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app

        client = TestClient(create_app(Path(tempfile.mkdtemp())))
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        pid = client.post("/api/projects", json={"name": "P", "goal": "g"}, headers=h).json()["id"]
        r = client.get(f"/api/projects/{pid}/data-value", headers=h)
        self.assertEqual((r.status_code, r.json()), (200, {}))
        r = client.post(f"/api/projects/{pid}/data-value", headers=h)
        self.assertEqual(r.status_code, 409, r.text)
        self.assertIn("models stage", r.json()["detail"])

    def test_a_second_request_while_one_is_queued_is_busy_not_an_error(self):
        """The review of 34.13: the busy table had no entry for this job, so a second request crashed (500) instead of 409."""
        import os

        from fastapi.testclient import TestClient

        from dclab_rnd.agentic.server import create_app
        from dclab_rnd.studio.store import ProjectStore

        with mock.patch.dict(os.environ, {"DCLAB_WORKER": "external"}):  # nothing runs the queued job: it stays queued
            client = TestClient(create_app(Path(tempfile.mkdtemp())))
            h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
            pid = client.post("/api/projects", json={"name": "P", "goal": "g"}, headers=h).json()["id"]
            with mock.patch.object(ProjectStore, "read_stage", return_value={"status": "completed"}):
                first = client.post(f"/api/projects/{pid}/data-value?wait=false", headers=h)
                second = client.post(f"/api/projects/{pid}/data-value?wait=false", headers=h)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 409, second.text)
        self.assertIn("learning curve", second.json()["detail"])

    def test_the_job_frees_the_project_when_the_measurement_fails(self):
        from dclab_rnd.jobs import handlers
        from dclab_rnd.studio import data_value

        state = {"running": None}

        class Projects:
            def update(self, pid, **values):
                state.update(values)
                return dict(state)

            def get(self, pid):
                return dict(state)

        class Services:
            projects = Projects()

        with mock.patch.object(data_value, "measure", side_effect=data_value.DataValueError("no folds")), \
                mock.patch.object(handlers, "checkpoint", lambda: None):
            with self.assertRaises(data_value.DataValueError):
                handlers._data_value_run(Services(), {"project_id": "p1"})
        self.assertIsNone(state["running"])

if __name__ == "__main__":
    unittest.main()
