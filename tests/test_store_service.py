"""Packages 32.24 to 32.31: the service's knowledge built from the store.

The records are shaped as the studio writes them (a stage record's ``status: completed``, ``started_at`` and
``primary_metric``; the leakage stage's declared columns and candidates; PostgreSQL rows of ``project``, ``stage_record`` and
``transition``). A kept project becomes a build judged by code: a declared leaking column used as a feature, or a holdout
read twice, is incorrect; a missing contract is unknown, never incorrect; a clean build with its export checked is correct.
A walk kept twice is one engagement; an outcome of the holdout's own read is not use; the catalog shows a figure only from
three builds in three workspaces; an item is accepted only by a person and flagged when its source changes; an outside
finding is refused when it copies its source, even in a short sentence, and links to the campaign that tests it.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
except ImportError as error:
    MISSING = str(error)

AT = "2026-10-20T10:00:00+00:00"


def _records(pid, *, holdout_uses=1, bar="met", moment="At order time, before dispatch", leak=False, metric="roc_auc", value=0.81,
             ci=(0.78, 0.84), baseline=0.5, export=True):
    """A project's records as the studio writes them: (project, {stage: record}, moves, walk)."""
    project = {"id": pid, "industry": "logistics and delivery", "holdout_uses": holdout_uses, "data": {"rows": 12000, "columns": [{"name": c} for c in "abcdefghi"]},
               "solution": {"task": "binary", "target": "late", "prediction_moment": moment},
               "exports": {"code": {"checked": True, "same": True, "value": value, "expected": value, "final_run": f"run-{pid}"}} if export else {}}
    stage = lambda name, at, evidence, **kw: {"stage": name, "status": "completed", "started_at": at, "completed_at": at, "elapsed_seconds": 2.0,
                                              "task": "binary", "primary_metric": metric, "evidence": evidence, **kw}
    stages = {
        "data": stage("data", "2026-10-20T10:05:00+00:00", {"column_roles": {"distance": "feature", "final_fare": "feature" if leak else "excluded"},
                                                             "risk_features": {"distance": ["missingness"]},
                                                             "recipe": {"choices": [{"field": "split", "value": "time", "why": "orders arrive in time"}]}}),
        "leakage": stage("leakage", "2026-10-20T10:06:00+00:00", {"declared_leakage_features": ["final_fare"],
                                                                   "heuristic_review_candidates": [{"feature": "final_fare", "reasons": ["declared_post_outcome_or_contested"]}]}),
        "models": stage("models", "2026-10-20T10:07:00+00:00", {"cv_protocol": "TimeSeriesSplit(5) on training rows"}),
        "final": stage("final", "2026-10-20T10:08:00+00:00", {"holdout_metrics": {metric: value}, "holdout_primary_metric_ci": {"point": value, "low": ci[0], "median": value, "high": ci[1], "draws": 40, "method": "iid row bootstrap"},
                                                               "holdout_uses_in_this_project": holdout_uses, "baselines": {"majority_class": {metric: baseline}},
                                                               "success_bar": {"verdict": bar, "bar": {"metric": metric}}}, run={"run": f"run-{pid}"}),
    }
    moves = [{"at": "2026-10-20T10:01:00+00:00", "move": "set_solution", "status": "allowed", "from": "WF-01", "to": "WF-02"},
             {"at": "2026-10-20T10:09:00+00:00", "move": "read_holdout", "status": "blocked", "failed_checks": ["holdout_read_once"]}]
    walk = {"id": f"walk:{pid}", "project": pid, "draft": None, "started": AT, "ended": "2026-10-21T10:00:00+00:00", "outcome": "reported",
            "steps": [{"at": "2026-10-20T10:00:00+00:00", "kind": "draft_event", "ref": "d:1", "detail": {"kind": "chat"}},
                      {"at": "2026-10-20T10:10:00+00:00", "kind": "move", "ref": "set_solution"},
                      {"at": "2026-10-21T10:00:00+00:00", "kind": "stage", "ref": "final"}]}  # a night away: not work
    return project, stages, moves, walk


def _keep_files(store, pid, ws="day", **kw):
    project, stages, moves, walk = _records(pid, **kw)
    store.keep("workflows", ws, f"projects/{pid}/project.json", json.dumps(project).encode(), consent="train")
    for name, body in stages.items():
        store.keep("workflows", ws, f"projects/{pid}/stages/{name}.json", json.dumps(body).encode(), consent="train")
    store.keep("workflows", ws, f"projects/{pid}/transitions.jsonl", "".join(json.dumps(m) + "\n" for m in moves).encode(), consent="train")
    store.keep("workflows", ws, f"walks/{pid}.json", json.dumps(walk).encode(), consent="train")


def _keep_db(store, pid, ws="pg"):
    """The same project as a PostgreSQL workspace's rows (database keeper: ``db/<table>.jsonl``)."""
    project, stages, moves, walk = _records(pid)
    store.keep("workflows", ws, "db/project.jsonl", (json.dumps({"id": pid, "workspace_id": ws, "name": "n", "industry": project["industry"], "doc": project}) + "\n").encode(),
               consent="train")
    store.keep("workflows", ws, "db/stage_record.jsonl", "".join(json.dumps({"project_id": pid, "stage": k, "record": v}) + "\n" for k, v in stages.items()).encode(),
               consent="train")
    store.keep("workflows", ws, "db/transition.jsonl", "".join(json.dumps({"id": i, "project_id": pid, "at": m["at"], "move": m["move"], "status": m["status"], "entry": m}) + "\n"
                                                              for i, m in enumerate(moves)).encode(), consent="train")
    store.keep("workflows", ws, f"walks/{pid}.json", json.dumps(walk).encode(), consent="train")


@unittest.skipIf(MISSING, MISSING)
class ServiceTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.store import Layout, Store

        self.tmp = Path(tempfile.mkdtemp())
        self.layout = Layout(self.tmp / "store")
        self.store = Store(self.layout)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _judged(self):
        from dclab_rnd.store import service, shapers

        shapers.shape(self.layout)
        return {b.id: (b, v) for b, v in service.builds(self.layout)}

    @staticmethod
    def _checks(verdict):
        return {c.name: c.status for c in verdict.checks}

    def test_a_clean_build_is_correct_and_its_parts_are_read_from_the_studios_records(self):
        _keep_files(self.store, "good")
        build, verdict = self._judged()["build:day:good"]
        self.assertEqual(set(self._checks(verdict).values()), {"passed"}, verdict.checks)
        self.assertEqual(verdict.verdict, "correct")
        self.assertEqual(build.context["size"], "medium")
        self.assertEqual(build.data["columns"], 9)
        self.assertEqual(build.recipe[0]["field"], "split")
        self.assertEqual(build.spent["people_minutes"], 40.0)  # 10 minutes, then a day capped at 30
        self.assertEqual(build.spent["chat_turns"], 1)
        self.assertEqual(build.result["holdout"], 0.81)
        self.assertIn("read_holdout: holdout_read_once", build.went_wrong)
        self.assertIn("used", build.unknown)

    def test_a_used_leaking_column_or_a_second_read_is_incorrect_and_a_missing_record_is_unknown(self):
        _keep_files(self.store, "leaky", leak=True)
        _keep_files(self.store, "reread", holdout_uses=2)
        _keep_files(self.store, "nomoment", moment="")
        _keep_files(self.store, "noexport", export=False)
        judged = self._judged()
        leaky = {c.name: c for c in judged["build:day:leaky"][1].checks}
        self.assertEqual(leaky["no_leaking_column"].status, "failed")
        self.assertIn("final_fare", leaky["no_leaking_column"].why)
        self.assertEqual(judged["build:day:leaky"][1].verdict, "incorrect")
        self.assertEqual(self._checks(judged["build:day:reread"][1])["holdout_read_once"], "failed")
        self.assertEqual(judged["build:day:reread"][1].verdict, "incorrect")
        self.assertEqual(self._checks(judged["build:day:nomoment"][1])["contract_before_scores"], "unknown")
        self.assertEqual(judged["build:day:nomoment"][1].verdict, "unknown")
        self.assertEqual(self._checks(judged["build:day:noexport"][1])["export_reproduces"], "unknown")

    def test_the_interval_must_clear_the_baseline_on_the_metrics_own_side(self):
        _keep_files(self.store, "wide", value=0.51, ci=(0.40, 0.62), baseline=0.50)  # the point beats it, the interval does not
        _keep_files(self.store, "mae", metric="mae", value=3.0, ci=(2.5, 3.5), baseline=5.0)  # lower is better: passes
        _keep_files(self.store, "maebad", metric="mae", value=6.0, ci=(5.5, 6.5), baseline=5.0)
        judged = self._judged()
        self.assertEqual(self._checks(judged["build:day:wide"][1])["beats_baselines_with_interval"], "failed")
        self.assertEqual(self._checks(judged["build:day:mae"][1])["beats_baselines_with_interval"], "passed")
        self.assertEqual(self._checks(judged["build:day:maebad"][1])["beats_baselines_with_interval"], "failed")

    def test_a_postgresql_workspace_gives_the_same_build(self):
        _keep_db(self.store, "pgp")
        judged = self._judged()
        self.assertIn("build:pg:pgp", judged)
        build, verdict = judged["build:pg:pgp"]
        self.assertEqual(verdict.verdict, "correct", verdict.checks)
        self.assertIn("read_holdout: holdout_read_once", build.went_wrong)

    def test_the_same_project_id_in_two_workspaces_is_two_builds_and_a_rewritten_log_is_counted_once(self):
        _keep_files(self.store, "same", ws="w1")
        _keep_files(self.store, "same", ws="w2")
        log = "".join(json.dumps({"at": AT, "purpose": "home_agent", "project_id": "same", "cost_eur": 0.25}) + "\n" for _ in range(2))
        self.store.keep_log("model_requests", "w1", "model_requests.jsonl", log.encode(), consent="train")
        self.store.keep_log("model_requests", "w1", "model_requests.jsonl", ("{\"rewritten\": true}\n" + log).encode(), consent="train")  # kept whole again
        judged = self._judged()
        self.assertIn("build:w1:same", judged)
        self.assertIn("build:w2:same", judged)
        self.assertEqual(judged["build:w1:same"][0].spent["eur"], 0.5)
        self.assertEqual(judged["build:w2:same"][0].spent["eur"], 0.0)

    def test_the_holdouts_own_outcome_is_not_use_and_a_business_outcome_is(self):
        _keep_files(self.store, "a")
        _keep_files(self.store, "b")
        events = [{"kind": "outcome.observed", "at": AT, "actor": "system", "ids": {"project": "a"}, "payload": {"observed": "holdout"}},
                  {"kind": "outcome.observed", "at": AT, "actor": "person", "ids": {"project": "b"}, "payload": {"observed": "business"}}]
        self.store.keep_log("sessions", "day", "experience/events.jsonl", "".join(json.dumps(e) + "\n" for e in events).encode(), consent="train")
        judged = self._judged()
        self.assertIsNone(judged["build:day:a"][0].used)
        self.assertTrue(judged["build:day:b"][0].used)

    def test_a_record_without_consent_to_improve_is_not_read(self):
        project, *_ = _records("op")
        self.store.keep("workflows", "op", "projects/op/project.json", json.dumps(project).encode(), consent="operate")
        self.assertNotIn("build:op:op", self._judged())

    def test_the_catalog_shows_a_figure_only_from_three_builds_in_three_workspaces(self):
        from dclab_rnd.store import service

        for pid in ("a", "b", "c"):
            _keep_files(self.store, pid, ws="acme")  # one business's three builds
        self._judged()
        self.assertFalse(service.service_catalog(self.layout)["binary"]["enough"])
        self.assertIn("workspaces", service.service_catalog(self.layout)["binary"]["why"])
        for ws in ("w2", "w3"):
            _keep_files(self.store, "x", ws=ws)
        self._judged()
        catalog = service.service_catalog(self.layout)["binary"]
        self.assertTrue(catalog["enough"])
        self.assertEqual(catalog["met_share"], 1.0)
        self.assertEqual(len(catalog["met_share_interval"]), 2)
        self.assertLess(catalog["met_share_interval"][0], 1.0)
        guide = service.guides(self.layout, folder=self.tmp / "accepted")["binary"]
        self.assertIn("[CATALOG]", guide)

    def test_a_student_refit_is_a_build_and_its_curve_says_too_few(self):
        from dclab_rnd.store import service

        card = {"point": "column.is_identifier", "name": "logistic", "version": "v1", "family": "logistic", "trained_on": 120, "groups": 9,
                "chosen_by": "out-of-fold accuracy on table-grouped folds", "dev": {"n": 40, "accuracy": 0.9, "accuracy_ci": [0.8, 0.96]},
                "rule_dev": {"n": 40, "accuracy": 0.7}}
        self.store.keep("students", "rnd", "students/column.is_identifier/logistic/v1/card.json", json.dumps(card).encode(), consent="train")
        self.store.keep("students", "rnd", "students/column.is_identifier/logistic/v1/measurement.json", json.dumps({"above_rule": True}).encode(), consent="train")
        judged = self._judged()
        build, verdict = judged["build:student:column.is_identifier/logistic/v1"]
        self.assertEqual(build.model_type, "m1_students")
        self.assertEqual(self._checks(verdict)["beats_baselines_with_interval"], "passed")
        curve = next(c for c in service.curves(self.layout) if c.subject == "column.is_identifier")
        self.assertFalse(curve.enough)

    def test_a_knowledge_item_is_only_proposed_until_a_person_accepts_it_and_flagged_when_its_source_changes(self):
        from dclab_rnd.store import service

        folder = self.tmp / "accepted"
        items = service.knowledge(folder=folder)
        self.assertTrue(any(i.id.startswith("CLAUDE-R") for i in items))
        self.assertTrue(all(i.accepted_by is None for i in items))
        with self.assertRaises(ValueError):
            service.accept(items[0].id, " ", folder=folder)
        path = service.accept(items[0].id, "owner", folder=folder)
        again = {i.id: i for i in service.knowledge(folder=folder)}
        self.assertEqual(again[items[0].id].accepted_by, "owner")
        self.assertIsNone(again[items[0].id].flagged)
        self.assertEqual(sum(1 for i in again.values() if i.accepted_by), 1)
        kept = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**kept, "practice": "an older wording"}), encoding="utf-8")
        self.assertTrue(service.knowledge(folder=folder)[0].flagged)
        # a guide cites only accepted, unflagged items
        service.accept(items[1].id, "owner", folder=folder)
        _keep_files(self.store, "a")
        self._judged()
        guide = service.guides(self.layout, folder=folder)["binary"]
        self.assertIn(f"[{items[1].id}]", guide)
        self.assertNotIn(f"[{items[0].id}]", guide)
        self.assertNotIn(f"[{items[2].id}]", guide)
        self.assertIn("Too few builds", guide)

    def test_an_outside_finding_needs_its_citation_never_copies_and_links_to_the_campaign_that_tests_it(self):
        from dclab_rnd.store import service

        source = ("Resampling the minority class changes the base rate the model sees, so its scores no longer match observed "
                  "frequencies and must be recalibrated on data with the true rate before any threshold is chosen.")
        ok = dict(citation="A. Author 2024", link="https://example.org/paper", licence="CC-BY-4.0", source_text=source)
        with self.assertRaises(ValueError):
            service.external(self.layout, summary="Calibrate after resampling.", **{**ok, "citation": ""})
        with self.assertRaises(ValueError):
            service.external(self.layout, summary="Calibrate after resampling.", **{**ok, "source_text": ""})
        with self.assertRaises(ValueError):  # a short sentence copied word for word
            service.external(self.layout, summary="its scores no longer match observed frequencies and must be recalibrated", **ok)
        item = service.external(self.layout, summary="After oversampling, fix the probabilities on untouched data before picking a cut-off.", **ok)
        self.assertEqual(item.status, "from_the_literature")
        self.assertIn(item.id, {i.id for i in service.knowledge(folder=self.tmp / "accepted", layout=self.layout)})
        confirmed = service.tested(self.layout, item.id, "CAL-001", True)
        self.assertEqual(confirmed.status, "measured_here")
        self.assertEqual(confirmed.evidence[-1]["campaign"], "CAL-001")
        contradicted = service.tested(self.layout, item.id, "CAL-002", False)
        self.assertIn("CAL-002", contradicted.counterexamples)
        self.assertTrue(contradicted.flagged)

    def test_a_walk_kept_twice_is_one_engagement_and_the_playbook_counts_what_it_rests_on(self):
        from dclab_rnd.store import service

        _keep_files(self.store, "a")
        draft = {"id": "walk:d9", "draft": "d9", "project": None, "outcome": "stopped_in_draft", "steps": [{"at": AT, "kind": "draft_event", "ref": "d9:1"}]}
        self.store.keep("workflows", "day", "walks/d8.json", json.dumps({**draft, "id": "walk:d8", "draft": "d8"}).encode(), consent="train")
        self.store.keep("workflows", "day", "walks/d8.json", json.dumps({**draft, "id": "walk:d8", "draft": "d8", "project": "a"}).encode(), consent="train")
        self.store.keep("workflows", "day", "walks/d9.json", json.dumps(draft).encode(), consent="train")
        self._judged()
        rows = {e.id: e for e in service.engagements(self.layout)}
        self.assertEqual(set(rows), {"engagement:day:a", "engagement:day:d8", "engagement:day:d9"})
        self.assertIsNone(rows["engagement:day:d8"].stopped)  # its newest version reached a project
        self.assertEqual(rows["engagement:day:a"].delivered["result"]["holdout"], 0.81)
        self.assertIn("draft_event", rows["engagement:day:d9"].stopped)
        self.assertIn("request", rows["engagement:day:d9"].unknown)
        book = {p.id: p for p in service.playbook(self.layout)}
        self.assertEqual(book["PB-failure-points"].evidence, ["engagement:day:d9"])
        self.assertFalse(book["PB-failure-points"].enough)
        written = service.write(self.layout)
        self.assertEqual(written["builds"], 1)
        self.assertEqual(written["engagements"], 3)
        self.assertTrue(self.layout.service("catalog").exists())

    def test_the_verdict_is_correct_only_when_every_check_passed(self):
        from dclab_core.service import CHECKS, ModelBuild, verdict

        build = ModelBuild(id="b", model_type="business_model", workspace="w", at=AT)
        self.assertEqual(verdict(build, {c: True for c in CHECKS}).verdict, "correct")
        self.assertEqual(verdict(build, {**{c: True for c in CHECKS}, "bar_met": None}).verdict, "unknown")
        self.assertEqual(verdict(build, {**{c: True for c in CHECKS}, "fitted_inside_folds": False, "bar_met": None}).verdict, "incorrect")


if __name__ == "__main__":
    unittest.main()
