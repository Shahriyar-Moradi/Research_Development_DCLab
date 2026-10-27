"""36.1: the coverage report and its gate.

The gate: every agent of the harness and every kind of help Home reads has a capture path (the events, decisions, model
purposes and store kinds it leaves) or a written reason why not; a new agent or track without one fails here.
The report: counts by agent, track and purpose, read from a tiny workspace built in the test.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.store import coverage as cov  # noqa: E402


class GateTest(unittest.TestCase):
    def test_every_agent_and_track_has_a_capture_path(self):
        self.assertEqual(cov.capture_gaps(), [])

    def test_a_new_agent_without_a_capture_path_fails(self):
        from dclab_rnd.agents import harness

        scratch = {**harness.AGENTS, "brand_new": SimpleNamespace(purposes=("home_agent",))}
        gaps = cov.capture_gaps(agents=scratch)
        self.assertTrue(any(g.startswith("agent brand_new") for g in gaps), gaps)

    def test_a_new_track_without_a_capture_path_fails(self):
        from dclab_rnd.draft import intent

        gaps = cov.capture_gaps(tracks=(*intent.TRACKS, "forecasting_bi"))
        self.assertTrue(any(g.startswith("track forecasting_bi") for g in gaps), gaps)

    def test_a_path_that_names_nothing_real_fails(self):
        bad = {**cov.CAPTURE, "home": cov.Capture(("event:run.recorded", "event:no.such.kind", "purpose:home_agent"))}
        gaps = cov.capture_gaps(capture=bad)
        self.assertTrue(any("no.such.kind" in g for g in gaps), gaps)

    def test_an_agent_that_names_none_of_its_purposes_fails(self):
        bad = {**cov.CAPTURE, "intern": cov.Capture(("event:run.recorded",))}
        self.assertTrue(any("purpose:intern" in g for g in cov.capture_gaps(capture=bad)))


class ReportTest(unittest.TestCase):
    def test_counts_from_a_tiny_workspace(self):
        home = Path(tempfile.mkdtemp())
        day = home / "model_exchanges" / "2026" / "10"
        day.mkdir(parents=True)
        lines = [{"model_request": "r1", "purpose": "home_agent", "whole": True, "thinking": [{"text": "Read the request."}], "messages": [{"role": "user", "content": "Predict churn"}]},
                 {"model_request": "r2", "purpose": "intern", "whole": False, "thought": True},
                 {"model_request": "r3", "purpose": "home_agent", "whole": True, "thinking": [], "messages": [{"role": "user", "content": "ریزش مشتری را پیش‌بینی کن"}]}]
        (day / "28.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n", encoding="utf-8")
        store = Path(tempfile.mkdtemp())
        from dclab_rnd.store.layout import Layout

        doc = cov.measure(Layout(store), home=home)
        by_purpose = doc["sources"]["by_purpose"]
        self.assertIn("home_agent", cov.render(doc))
        self.assertEqual((by_purpose["home_agent"]["whole"], by_purpose["home_agent"]["thinking"]), (2, 1))
        self.assertEqual(by_purpose["intern"]["thought_lost"], 1)  # thought, but only the shape was kept
        self.assertEqual((by_purpose["home_agent"]["fa"], by_purpose["home_agent"]["en"]), (1, 1))
        self.assertIsInstance(cov.gaps(doc), list)
        summary = cov.summary(doc)
        self.assertIsInstance(summary, dict)


def gz(path, rows):
    import gzip

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r) for r in rows) + "\n")


class StoreCountsTest(unittest.TestCase):
    """The review's scenarios on a tiny store: one request once, runs and tracks attributed by the rules."""

    def setUp(self):
        from dclab_rnd.store.layout import Layout

        self.layout = Layout(Path(tempfile.mkdtemp()))

    def test_a_request_kept_in_two_entry_versions_counts_once(self):
        line = {"purpose": "home_agent", "model": "m", "at": "2026-10-28T09:00:00+00:00", "draft": "d1"}
        prov = {"source": "model_requests.jsonl", "workspace": "w1"}
        gz(self.layout.records_of("model_requests") / "a.jsonl.gz",
           [{"id": "model_requests:w1:v1#1", "provenance": prov, "body": {**line, "_context": "v1"}},
            {"id": "model_requests:w1:v11#1", "provenance": prov, "body": {**line, "_context": "v11"}},
            {"id": "db:1", "provenance": {"source": "db/model_request.jsonl", "workspace": "w2"}, "body": {**line, "id": "q9"}}])
        doc = cov.measure(self.layout, home=Path(tempfile.mkdtemp()))
        self.assertEqual(doc["sources"]["by_purpose"]["home_agent"]["calls"], 2)  # the file line once, the database row once

    def test_runs_by_agent_and_track(self):
        events = [{"body": {"kind": "run.recorded", "ids": {"conversation": "s1"}, "payload": {"run_kind": "agent", "run": "r1", "name": "intern"}}},
                  {"body": {"kind": "run.recorded", "ids": {"conversation": "d7"}, "payload": {"run_kind": "agent", "run": "r2", "name": "home_orchestrator"}}},
                  {"body": {"kind": "draft.built", "ids": {"conversation": "d7", "project": "p7"}, "payload": {}}}]
        gz(self.layout.records_of("sessions") / "s.jsonl.gz", events)
        doc = cov.measure(self.layout, home=Path(tempfile.mkdtemp()))
        by_agent, by_track = doc["sources"]["by_agent"], doc["sources"]["by_track"]
        self.assertEqual(by_agent["home.project"]["runs"], 1)  # the orchestrator's runs under its agent, not its purpose
        self.assertNotIn("home_orchestrator", by_agent)
        self.assertEqual(by_track[cov.NO_REQUEST]["runs"], 1)  # an intern session is no draft
        self.assertEqual(by_track["prediction"]["runs"], 1)    # a built draft read before H2

    def test_only_the_persons_own_label_moves_a_draft(self):
        t = cov._Tracks()
        t.decision({"point": "request.intent", "id": "rec1", "draft": "d1", "value_used": "prediction"})
        t.labelled({"record": "rec1", "value": "analysis", "source": "blind"})
        self.assertEqual(t.of("d1"), "prediction")
        t.labelled({"record": "rec1", "value": "analysis", "source": "user_decision_exposed"})
        self.assertEqual(t.of("d1"), "analysis")


class CatalogTest(unittest.TestCase):
    def test_a_rebuild_keeps_the_summary_and_the_copy_holds_no_time(self):
        from dclab_rnd.store import catalog
        from dclab_rnd.store.layout import Layout

        layout = Layout(Path(tempfile.mkdtemp()))
        copy = Path(tempfile.mkdtemp()) / "STORE_CATALOG.json"
        doc = cov.measure(layout, home=Path(tempfile.mkdtemp()))
        cov.write(layout, doc, copy=copy)
        catalog.write(layout, copy=copy) if "copy" in catalog.write.__code__.co_varnames else catalog.write(layout)
        held = json.loads(layout.catalog.read_text())
        self.assertIn("coverage", held)
        if copy.exists():
            self.assertNotIn("built_at", json.loads(copy.read_text()).get("coverage", {}))


if __name__ == "__main__":
    unittest.main()
