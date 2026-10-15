"""23.4: every error carries a fingerprint (the exception's type and the top frame of our own code), a failed stage and a
failed job write one under their ids, a limit hit is an error of its own, and the record groups the last days' errors
by fingerprint for the Sessions page and the command line."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient

    from dclab_core import errors
    from dclab_rnd import experience
    from dclab_rnd.experience import finder
except ImportError as error:  # the agent environment may lack pydantic
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class FingerprintTest(unittest.TestCase):
    def test_the_same_failure_is_one_fingerprint_and_a_different_frame_another(self):
        a = errors.fingerprint("ValueError", [("dclab_core.consent", "exportable"), ("tests.x", "test")])
        b = errors.fingerprint("ValueError", [("dclab_core.consent", "exportable"), ("tests.y", "other")])
        c = errors.fingerprint("ValueError", [("dclab_core.card", "problem_card")])
        self.assertEqual(a, b)
        self.assertNotEqual(a["fingerprint"], c["fingerprint"])
        self.assertEqual(a["frame"], "dclab_core.consent:exportable")
        outside = errors.fingerprint("KeyError", [("json.decoder", "decode"), ("tests.x", "test")])
        self.assertEqual((outside["frame"], len(outside["fingerprint"])), (None, 12))  # a library's error groups by its type alone
        host = errors.fingerprint("KeyError", [("json.decoder", "decode"), ("app.services.x", "run")], own=("app",))
        self.assertEqual(host["frame"], "app.services.x:run")  # a host names its own packages

    def test_a_caught_exception_names_the_innermost_frame_of_ours(self):
        from dclab_core import consent

        try:
            consent.exportable("train", "sell")
        except ValueError as exc:
            found = errors.from_exception(exc)
        self.assertEqual((found["code"], found["frame"]), ("ValueError", "dclab_core.consent:exportable"))
        self.assertEqual(found["fingerprint"], errors.fingerprint("ValueError", [("dclab_core.consent", "exportable")])["fingerprint"])
        self.assertNotIn("sell", found["fingerprint"] + (found["frame"] or ""))  # nothing of the message enters


@unittest.skipIf(MISSING, MISSING)
class GroupingTest(unittest.TestCase):
    def test_the_last_days_errors_group_by_fingerprint(self):
        log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "w1")
        same = errors.fingerprint("ValueError", [("dclab_rnd.studio.engine", "execute")])
        for i in range(3):
            experience.record("error.raised", "system", {"where": "job:stage", "code": "ValueError", **same}, log=log, visit=f"v{i}", request=f"r{i}", job=f"j{i}")
        experience.record("stage.failed", "system", {"stage": "data", "code": "ValueError", **same}, log=log, visit="v0", request="r0", job="j0")  # the same failure, caught twice
        experience.record("stage.failed", "system", {"stage": "final", "code": "KeyError", **errors.fingerprint("KeyError", [])}, log=log, visit="v9")
        experience.record("error.raised", "system", {"where": "intern", "code": "Old"}, log=log, visit="v9")  # no fingerprint: by kind and code
        groups = finder.errors(log, days=30)
        self.assertEqual([g["count"] for g in groups], [3, 1, 1])
        top = groups[0]
        self.assertEqual((top["fingerprint"], top["frame"], top["kinds"], top["codes"], top["visits"]),
                         (same["fingerprint"], "dclab_rnd.studio.engine:execute", ["error.raised", "stage.failed"], ["ValueError"], 3))
        self.assertEqual((sorted(top["requests"]), sorted(top["jobs"])), (["r0", "r1", "r2"], ["j0", "j1", "j2"]))  # one job, one count
        self.assertIn("error.raised:Old", [g["fingerprint"] for g in groups])
        from dclab_rnd.experience import __main__ as cli

        with mock.patch.object(cli, "_log", return_value=log):
            import contextlib
            import io

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(cli.main(["errors", "--days", "30"]), 0)
            self.assertIn(same["fingerprint"], out.getvalue())
            self.assertIn("r0", out.getvalue())


@unittest.skipIf(MISSING, MISSING)
class ProductTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_rnd.agentic.server import create_app

        cls.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        cls.services = cls.client.app.state.services
        cls.h = {"X-DCLab-Token": cls.client.get("/api/config").json()["csrf"]}

    def test_a_failed_stage_carries_a_fingerprint_and_the_page_groups_it(self):
        from dclab_rnd import context
        from dclab_rnd.studio import engine

        pid = self.services.projects.create("Breaks", "general", "Predict churn")["id"]
        with context.using(self.services), mock.patch.object(engine, "prepare", return_value=mock.MagicMock(spec=[])), \
                mock.patch.object(engine, "stage_data", side_effect=RuntimeError("the stage broke")):
            try:  # past the graph's gate and the data (not the point here): the stage runner's own failure branch
                engine._execute(self.services.projects, pid, "data", self.services.projects.get(pid), None)
            except Exception:  # noqa: BLE001 - the failure is the point
                pass
        failed = [e for e in self.services.experience.read(project=pid) if e.kind == "stage.failed"]
        self.assertTrue(failed, "no stage.failed event")
        self.assertEqual(failed[-1].payload["code"], "RuntimeError")
        self.assertTrue(failed[-1].payload["fingerprint"] and failed[-1].payload["frame"].startswith("dclab_rnd.studio.engine:"))
        self.assertEqual(self.services.projects.get(pid)["stages"]["data"]["fingerprint"], failed[-1].payload["fingerprint"])
        groups = self.client.get("/api/experience/errors", params={"days": 7}).json()["groups"]
        hit = next(g for g in groups if g["fingerprint"] == failed[-1].payload["fingerprint"])
        self.assertEqual((hit["kinds"], hit["codes"], hit["where"]), (["stage.failed"], ["RuntimeError"], ["data"]))
        self.assertEqual(self.client.get("/api/experience/errors", params={"days": 0}).status_code, 422)

    def test_a_failed_job_writes_the_error_under_its_request_within_the_run(self):
        from dclab_rnd.jobs import worker as jw

        store = self.services.job_store

        class Boom:
            kind = "boom"

            def run(self, env, payload):
                raise ValueError("the job broke on purpose")

        with experience.bind(visit="visit-j", request="req-j"):
            job = store.enqueue("boom", "k1", {"project_id": None})
        with mock.patch.dict(jw.HANDLERS, {"boom": jw.Handler(Boom().run, None)}, clear=False):
            w = jw.Worker(store, self.services, threads=1, poll=0.01)
            w.execute(store.claim(w.id))
        raised = [e for e in self.services.experience.read(kinds=["error.raised"]) if e.payload.get("where") == "job:boom"]
        self.assertTrue(raised, "no error.raised for the failed job")
        self.assertEqual((raised[-1].ids.request, raised[-1].ids.job, raised[-1].payload["code"]), ("req-j", job["id"], "ValueError"))
        self.assertEqual(raised[-1].payload["frame"], "dclab_rnd.jobs.worker:execute")  # the handler is not ours: the innermost frame of ours


if __name__ == "__main__":
    unittest.main()
