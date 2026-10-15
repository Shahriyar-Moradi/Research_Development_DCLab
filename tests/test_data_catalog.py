"""23.9: the catalog lists everything the code records, and the gate fails when it drifts; the standard questions answer on a
seeded record; a request id opens as a timeline on the log, the route and the command line."""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient

    from dclab_core import experience as x, ledger as L, outcomes as oc
    from dclab_rnd import data_catalog, experience, ledger
    from dclab_rnd.experience import queries
except ImportError as error:  # the agent environment may lack pydantic
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class CatalogTest(unittest.TestCase):
    def test_the_committed_catalog_is_what_the_code_generates(self):
        stale = [p.name for p, text in data_catalog.outputs().items() if not p.exists() or p.read_text(encoding="utf-8") != text]
        self.assertEqual(stale, [], "run: make catalog")

    def test_it_lists_every_kind_point_audit_kind_builder_and_table(self):
        from dclab_core import datasets
        from dclab_rnd import audit
        from dclab_rnd.storage.models import metadata
        from typing import get_args

        doc = data_catalog.build()
        self.assertEqual([k["kind"] for k in doc["event_kinds"]], sorted(x.KINDS))
        self.assertEqual({p["point"] for p in doc["decision_points"]}, set(L.POINTS))
        self.assertEqual(doc["audit_kinds"], list(audit.KINDS))
        self.assertEqual({b["builder"] for b in doc["dataset_builders"]}, set(get_args(datasets.Builder)))
        self.assertEqual({t["table"] for t in doc["tables"]}, set(metadata.tables))
        by = {k["kind"]: k for k in doc["event_kinds"]}
        self.assertEqual((by["model.request"]["never_leaves"], by["model.request"]["train_only"]), ([], ["prompt", "reply"]))  # model text: train only (23.10)
        self.assertEqual(by["tool.call"]["never_leaves"], ["arguments", "result"])    # a tool's values too
        self.assertIn("dclab_rnd/experience/hooks.py", by["stage.failed"]["emitted_by"])
        self.assertEqual(by["feedback.given"]["leaves_at"], "improve")

    def test_a_kind_the_code_records_is_in_the_vocabulary(self):
        """Every literal kind passed to record() in the studio is a kind of the contract (and so in the catalog)."""
        pattern = re.compile(r'\brecord\(\s*"([a-z_]+\.[a-z_]+)"')
        found = {}
        for path in (ROOT / "dclab_rnd").rglob("*.py"):
            for kind in pattern.findall(path.read_text(encoding="utf-8")):
                found.setdefault(kind, path.relative_to(ROOT).as_posix())
        self.assertTrue(found)
        self.assertEqual({k: p for k, p in found.items() if k not in x.KINDS}, {}, "a kind recorded but not in dclab_core.experience.KINDS")

    def test_an_undescribed_table_or_builder_fails_the_catalog(self):
        with mock.patch.dict(data_catalog.NOTES, {"builder:outcomes": ""}):
            with self.assertRaises(SystemExit):
                data_catalog.build()

    def test_check_fails_on_drift_and_passes_when_fresh(self):
        with mock.patch.object(data_catalog, "DOCS", Path(tempfile.mkdtemp())):
            self.assertEqual(data_catalog.main(["--check"]), 1)  # nothing written yet
            self.assertEqual(data_catalog.main([]), 0)
            self.assertEqual(data_catalog.main(["--check"]), 0)
            (data_catalog.DOCS / "DATA_CATALOG.md").write_text("old", encoding="utf-8")
            self.assertEqual(data_catalog.main(["--check"]), 1)


@unittest.skipIf(MISSING, MISSING)
class QueriesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_rnd.agentic.server import create_app

        cls.client = TestClient(create_app(Path(tempfile.mkdtemp())))
        cls.client.__enter__()
        cls.s = cls.client.app.state.services
        cls.h = {"X-DCLab-Token": cls.client.get("/api/config").json()["csrf"]}
        log, book = cls.s.experience, cls.s.ledger
        cls.project = cls.s.projects.create("Asked", "general", "Predict churn")["id"]
        p = cls.project
        record = lambda kind, actor, payload, **ids: experience.record(kind, actor, payload, log=log, project=p, **ids)  # noqa: E731
        record("stage.finished", "system", {"stage": "data", "metric": "roc_auc", "value": 0.8}, visit="v1", request="req-0001", run="run-1")
        record("graph.move", "system", {"move": "start_final", "allowed": False, "failed_checks": ["WF-04"]}, visit="v1", request="req-0001")
        record("graph.move", "system", {"move": "start_final", "allowed": False, "failed_checks": ["WF-04"]}, visit="v1", request="req-0002")
        record("decision.refused", "system", {"point": "split.strategy", "choice": "random", "rule": "time-ordered"}, visit="v1", request="req-0002")
        record("model.request", "model", {"purpose": "home_agent", "cost": 0.5, "currency": "EUR", "outcome": "ok"}, visit="v1", request="req-0003")
        record("model.request", "model", {"purpose": "home_agent", "cost": 0.0, "currency": "EUR", "outcome": "error: Timeout"}, visit="v1", request="req-0004")
        a = lambda by, value: L.Answer(by=by, answerer=by, value=value)  # noqa: E731
        cls.rec = L.DecisionRecord(id="dr000001", point="split.strategy", subject=L.Subject(kind="project", key=p), at="2026-10-16T10:00:00+00:00",
                                   answers=[a("rule", "time"), a("person", "random")], value_used="random", used="person", project=p, consent="train")
        book.add(cls.rec)
        book.add_outcomes([oc.Outcome(id="do000001", record="dr000001", point="split.strategy", metric="roc_auc", value=0.7, reference=0.8,
                                    higher_is_better=True, observed="holdout", verdict="badly", by="final_stage", at="2026-10-16T11:00:00+00:00", project=p)])

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def ask(self, name, **args):
        return queries.run(name, self.s.experience, self.s.ledger, **args)

    def test_what_happened_in_a_request(self):
        got = self.ask("request", request="req-0002")
        self.assertEqual([r[1] for r in got["rows"]], ["graph.move", "decision.refused"])
        self.assertEqual(got["columns"][:2], ["at", "kind"])
        with self.assertRaises(ValueError):
            self.ask("request")

    def test_where_agents_are_rejected_most_and_which_calls_failed(self):
        got = self.ask("rejections", days=30)
        self.assertEqual(got["rows"][0][:4], ["graph.move", "start_final", "WF-04", 2])
        self.assertIn(["decision.refused", "split.strategy", "time-ordered", 1], [r[:4] for r in got["rows"]])
        self.assertEqual(self.ask("failed_calls")["rows"], [["home_agent", "error: Timeout", 1, 2]])

    def test_which_decisions_were_overridden_and_did_approved_ones_turn_out_well(self):
        self.assertEqual(self.ask("overrides")["rows"], [["split.strategy", "time", "random", 1]])
        got = self.ask("outcomes")
        self.assertEqual([r for r in got["rows"] if r[0] == "split.strategy"], [["split.strategy", 0, 1, 0, 0]])

    def test_cost_per_week_what_is_ready_for_training_and_what_a_deletion_removes(self):
        cost = self.ask("cost")
        self.assertEqual([r[1:] for r in cost["rows"]], [["EUR", 0.5, 2]])
        self.assertRegex(cost["rows"][0][0], r"^\d{4}-W\d{2}$")
        training = {(r[0], r[1]): r[2] for r in self.ask("training")["rows"]}
        self.assertEqual(training[("decisions", "train")], 1)
        deletion = dict(self.ask("deletion", project=self.project)["rows"])
        self.assertEqual((deletion["decisions"], deletion["outcomes"]), (1, 1))
        self.assertGreaterEqual(deletion["events"], 6)
        with self.assertRaises(ValueError):
            self.ask("deletion")

    def test_a_project_story_lists_its_moves_and_counts_its_decisions(self):
        rows = self.ask("project", project=self.project)["rows"]
        self.assertIn("stage.finished", [r[1] for r in rows])
        self.assertIn("1 decisions at 1 points; 1 outcome rows", rows[-1][3])

    def test_the_pages_route_the_same_questions_and_a_request_id_opens_as_a_timeline(self):
        got = self.client.get("/api/experience/queries/rejections", params={"days": 30}).json()
        self.assertEqual(got["rows"][0][3], 2)
        self.assertEqual(self.client.get("/api/experience/queries/nonsense").status_code, 422)
        self.assertEqual(self.client.get("/api/experience/queries/request").status_code, 422)  # names no request
        self.assertEqual(self.client.get("/api/experience/queries/cost", params={"days": 999}).status_code, 422)
        found = self.client.get("/api/experience/search", params={"request": "req-0001"}).json()
        self.assertEqual({e["ids"].get("request") for e in found["events"]}, {"req-0001"})
        self.assertEqual(sorted(e["kind"] for e in found["events"]), ["graph.move", "stage.finished"])
        self.assertEqual(self.client.get("/api/experience/search", params={"request": "bad id!"}).status_code, 422)

    def test_the_log_filters_by_request_on_files(self):
        log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "w1")
        experience.record("stage.finished", "system", {"stage": "data"}, log=log, request="r-a", visit="v1")
        experience.record("stage.finished", "system", {"stage": "final"}, log=log, request="r-b", visit="v1")
        self.assertEqual([e.payload["stage"] for e in log.read(request="r-b")], ["final"])



@unittest.skipIf(MISSING, MISSING)
class PortableTest(unittest.TestCase):
    """The questions are pure functions over lists, so a host with other storage asks them of its adapted rows."""

    def test_the_core_answers_over_plain_lists(self):
        from dclab_core import catalog, queries as q

        events = [x.make("model.request", "model", {"workspace": "w"}, {"purpose": "p", "cost": 0.25, "currency": "USD", "outcome": "ok"}, id="ev000001", at="2026-10-15T10:00:00+00:00"),
                  x.make("model.request", "model", {"workspace": "w"}, {"purpose": "p", "cost": 0.0, "currency": "USD", "outcome": "error: Timeout"}, id="ev000002", at="2026-10-15T10:00:01+00:00"),
                  x.make("graph.move", "system", {"workspace": "w", "request": "r1"}, {"move": "m", "allowed": False, "failed_checks": ["WF-01"]}, id="ev000003", at="2026-10-15T10:00:02+00:00")]
        self.assertEqual(q.failed_calls(events)["rows"], [["p", "error: Timeout", 1, 2]])
        self.assertEqual(q.cost(events)["rows"][0][1:], ["USD", 0.25, 2])
        self.assertEqual(q.rejections(events)["rows"][0][:4], ["graph.move", "m", "WF-01", 1])
        self.assertEqual([r[1] for r in q.request(events, "r1")["rows"]], ["graph.move"])
        self.assertEqual(q.training(events)["rows"], [["events", "operate", 3]])
        self.assertEqual(q.deletion(events, project_id="p")["rows"][0], ["events", 0])
        built = catalog.build({"graph.move": ["x.py"]})
        self.assertEqual({k["kind"] for k in built["event_kinds"]}, set(x.KINDS))
        self.assertEqual(next(k for k in built["event_kinds"] if k["kind"] == "graph.move")["emitted_by"], ["x.py"])
        self.assertEqual(built["dclab_core"], __import__("dclab_core").__version__)


@unittest.skipIf(MISSING, MISSING)
class ReviewTest(unittest.TestCase):
    def test_a_deletion_counts_what_forget_removes_not_the_newest_slice(self):
        from dclab_rnd.experience import retention

        home = Path(tempfile.mkdtemp())
        log = experience.FileLog(home / "experience", "w1")
        book = ledger.FileLedger(home / "ledger", "w1")
        for i in range(7):
            experience.record("stage.finished", "system", {"stage": "data"}, log=log, project="old", visit=f"v{i}")
        for i in range(7):
            experience.record("stage.finished", "system", {"stage": "data"}, log=log, project="new", visit="vn")
        with mock.patch.object(queries, "LIMIT", 5):  # a cap far below the record: the answer must not depend on it
            counted = dict(queries.run("deletion", log, book, project="old")["rows"])
        gone = retention.delete_events(log, project="old")
        self.assertEqual((counted["events"], gone), (7, 7))

    def test_it_counts_the_prompts_a_deletion_clears_and_the_replies_that_failed_their_check(self):
        from dclab_rnd import models

        home = Path(tempfile.mkdtemp())
        log = experience.FileLog(home / "experience", "w1")
        usage = models.FileUsage(home / "u.jsonl")
        usage.record({"at": "2026-10-16T10:00:00+00:00", "purpose": "home_agent", "tier": "t", "model": "m", "project_id": "p9", "prompt": "secret words", "outcome": "ok"})
        usage.record({"at": "2026-10-16T10:00:01+00:00", "purpose": "home_agent", "tier": "t", "model": "m", "project_id": "p9", "prompt": None, "outcome": "ok"})
        self.assertEqual(usage.count_prompts(project_id="p9"), 1)
        rows = dict(queries.run("deletion", log, ledger.FileLedger(home / "ledger", "w1"), project="p9", usage=usage)["rows"])
        self.assertEqual(rows["prompts kept in the usage log"], 1)
        self.assertEqual(usage.forget_prompts(project_id="p9"), 1)  # what it counted is what it clears
        for i in range(2):
            usage.record_check({"at": "2026-10-16T10:00:0%d+00:00" % i, "purpose": "stage_notes", "tier": "t", "model": "m", "passed": False, "reason": "no source cited"})
        got = queries.run("failed_calls", log, None, usage=usage, days=365)
        self.assertEqual(got["rows"], [["stage_notes", "failed its check: no source cited", 2, 2]])

    def test_a_request_id_a_proxy_chose_opens_and_a_token_label_never_shows(self):
        from dclab_core import queries as q

        client = TestClient(__import__("dclab_rnd.agentic.server", fromlist=["create_app"]).create_app(Path(tempfile.mkdtemp())))
        with client:
            s = client.app.state.services
            experience.record("stage.finished", "system", {"stage": "data"}, log=s.experience, request="svc.7f3a9c01", visit="v1")
            ok = client.get("/api/experience/queries/request", params={"request": "svc.7f3a9c01"})
            self.assertEqual((ok.status_code, len(ok.json()["rows"])), (200, 1))
            self.assertEqual(client.get("/api/experience/search", params={"request": "svc.7f3a9c01"}).status_code, 200)
            self.assertEqual(client.get("/api/experience/search", params={"request": "bad id!"}).status_code, 422)
        token = x.make("auth.token_created", "person", {"workspace": "w", "request": "r9"}, {"token": "tk1", "name": "Maryam's laptop"}, id="ev000009", at="2026-10-16T10:00:00+00:00")
        self.assertNotIn("Maryam", q.request([token], "r9")["rows"][0][5])  # a label a person typed is not meta

    def test_an_overridden_decision_ignores_a_shadow_answer_and_the_catalog_names_emitters_not_readers(self):
        a = lambda by, value, source=None: L.Answer(by=by, answerer=by, value=value, source=source)  # noqa: E731
        from dclab_core import queries as q

        rec = L.DecisionRecord(id="dr000002", point="split.strategy", subject=L.Subject(kind="project", key="p"), at="2026-10-16T10:00:00+00:00",
                               answers=[a("person", "random"), a("classifier", "time", "shadow")], value_used="random", used="person")
        self.assertEqual(q.overrides([rec])["rows"], [])  # only a shadow disagreed: it never answered
        by = {k["kind"]: k for k in data_catalog.build()["event_kinds"]}
        self.assertIn("dclab_rnd/experience/hooks.py", by["graph.move"]["emitted_by"])  # the emitter, not the adapter or the queries
        self.assertNotIn("dclab_core/adapters/main_product_experience.py", by["graph.move"]["emitted_by"])
        self.assertNotIn("dclab_core/queries.py", by["graph.move"]["emitted_by"])


if __name__ == "__main__":
    unittest.main()
